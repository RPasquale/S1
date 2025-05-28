"""
FastAPI server for the Data Extraction API

This provides REST endpoints to manage and query the data extraction system.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from data_extraction_api import DataExtractionAPI

app = FastAPI(
    title="Data Extraction API Server",
    description="API for extracting and managing data from agent architecture components",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API instance
extraction_api = DataExtractionAPI()

# Pydantic models
class ExtractionStats(BaseModel):
    total_items: int
    source_stats: Dict[str, Dict[str, Any]]
    type_stats: Dict[str, int]
    last_generated: str

class SearchQuery(BaseModel):
    query: str
    source_name: Optional[str] = None
    data_type: Optional[str] = None
    limit: int = 10

class SearchResult(BaseModel):
    source_name: str
    data_type: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str
    url: str

class ExtractionResponse(BaseModel):
    success: bool
    message: str
    items_extracted: int


@app.get("/", summary="API Health Check")
async def root():
    """Health check endpoint"""
    return {
        "message": "Data Extraction API is running",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/stats", response_model=ExtractionStats, summary="Get Extraction Statistics")
async def get_stats():
    """Get comprehensive statistics about extracted data"""
    try:
        stats = extraction_api.get_extraction_stats()
        return ExtractionStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.post("/extract/run", response_model=ExtractionResponse, summary="Run Extraction Cycle")
async def run_extraction(background_tasks: BackgroundTasks):
    """Run a full extraction cycle for all sources"""
    try:
        # Run extraction in background to avoid timeout
        background_tasks.add_task(extraction_api.run_extraction_cycle)
        return ExtractionResponse(
            success=True,
            message="Extraction cycle started in background",
            items_extracted=0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting extraction: {str(e)}")


@app.post("/extract/run-sync", response_model=ExtractionResponse, summary="Run Extraction Cycle (Synchronous)")
async def run_extraction_sync():
    """Run a full extraction cycle synchronously (may take time)"""
    try:
        items_extracted = extraction_api.run_extraction_cycle()
        return ExtractionResponse(
            success=True,
            message="Extraction cycle completed",
            items_extracted=items_extracted
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running extraction: {str(e)}")


@app.post("/search", response_model=List[SearchResult], summary="Search Extracted Data")
async def search_data(search_query: SearchQuery):
    """Search through extracted data with filters"""
    try:
        results = extraction_api.search_extracted_data(
            query=search_query.query,
            source_name=search_query.source_name,
            data_type=search_query.data_type,
            limit=search_query.limit
        )
        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching data: {str(e)}")


@app.get("/search", response_model=List[SearchResult], summary="Search Extracted Data (GET)")
async def search_data_get(
    q: str = Query(..., description="Search query"),
    source: Optional[str] = Query(None, description="Filter by source name"),
    type: Optional[str] = Query(None, description="Filter by data type"),
    limit: int = Query(10, description="Maximum number of results")
):
    """Search through extracted data using GET parameters"""
    try:
        results = extraction_api.search_extracted_data(
            query=q,
            source_name=source,
            data_type=type,
            limit=limit
        )
        return [SearchResult(**result) for result in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching data: {str(e)}")


@app.get("/sources", summary="Get Data Sources")
async def get_sources():
    """Get all configured data sources"""
    try:
        sources = extraction_api.db.get_data_sources()
        return [
            {
                "name": source.name,
                "url": source.url,
                "source_type": source.source_type,
                "extraction_method": source.extraction_method,
                "update_frequency": source.update_frequency,
                "last_updated": source.last_updated.isoformat() if source.last_updated else None,
                "enabled": source.enabled
            }
            for source in sources
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sources: {str(e)}")


@app.post("/export", summary="Export Data")
async def export_data(background_tasks: BackgroundTasks, output_dir: str = "exported_data"):
    """Export all extracted data to files"""
    try:
        background_tasks.add_task(extraction_api.export_data, output_dir)
        return {
            "success": True,
            "message": f"Data export started to {output_dir}",
            "output_directory": output_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting export: {str(e)}")


@app.get("/data-types", summary="Get Available Data Types")
async def get_data_types():
    """Get all available data types in the system"""
    try:
        import sqlite3
        conn = sqlite3.connect(extraction_api.db.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT data_type FROM extracted_data ORDER BY data_type")
        data_types = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        return {"data_types": data_types}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting data types: {str(e)}")


@app.get("/sources/{source_name}/stats", summary="Get Source-Specific Statistics")
async def get_source_stats(source_name: str):
    """Get statistics for a specific source"""
    try:
        import sqlite3
        conn = sqlite3.connect(extraction_api.db.db_path)
        cursor = conn.cursor()
        
        # Check if source exists
        cursor.execute("SELECT COUNT(*) FROM extracted_data WHERE source_name = ?", (source_name,))
        count = cursor.fetchone()[0]
        
        if count == 0:
            raise HTTPException(status_code=404, detail=f"Source '{source_name}' not found")
        
        # Get detailed stats
        cursor.execute("""
            SELECT 
                data_type,
                COUNT(*) as count,
                MIN(timestamp) as first_extraction,
                MAX(timestamp) as last_extraction,
                AVG(LENGTH(content)) as avg_content_length
            FROM extracted_data 
            WHERE source_name = ?
            GROUP BY data_type
            ORDER BY count DESC
        """, (source_name,))
        
        type_stats = {}
        for row in cursor.fetchall():
            type_stats[row[0]] = {
                "count": row[1],
                "first_extraction": row[2],
                "last_extraction": row[3],
                "avg_content_length": int(row[4]) if row[4] else 0
            }
        
        # Get total stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_items,
                MIN(timestamp) as first_extraction,
                MAX(timestamp) as last_extraction,
                SUM(LENGTH(content)) as total_content_size
            FROM extracted_data 
            WHERE source_name = ?
        """, (source_name,))
        
        total_row = cursor.fetchone()
        
        conn.close()
        
        return {
            "source_name": source_name,
            "total_items": total_row[0],
            "first_extraction": total_row[1],
            "last_extraction": total_row[2],
            "total_content_size": total_row[3],
            "type_breakdown": type_stats
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting source stats: {str(e)}")


@app.get("/recent", response_model=List[SearchResult], summary="Get Recent Extractions")
async def get_recent_extractions(
    limit: int = Query(20, description="Number of recent items to return"),
    source: Optional[str] = Query(None, description="Filter by source name")
):
    """Get the most recently extracted data"""
    try:
        import sqlite3
        conn = sqlite3.connect(extraction_api.db.db_path)
        cursor = conn.cursor()
        
        sql = """
            SELECT source_name, data_type, content, metadata, timestamp, url
            FROM extracted_data
        """
        params = []
        
        if source:
            sql += " WHERE source_name = ?"
            params.append(source)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        results = []
        
        for row in cursor.fetchall():
            import json
            results.append(SearchResult(
                source_name=row[0],
                data_type=row[1],
                content=row[2][:500] + "..." if len(row[2]) > 500 else row[2],
                metadata=json.loads(row[3]),
                timestamp=row[4],
                url=row[5]
            ))
        
        conn.close()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent extractions: {str(e)}")


@app.get("/logs", summary="Get Extraction Logs")
async def get_extraction_logs(
    limit: int = Query(50, description="Number of log entries to return"),
    source: Optional[str] = Query(None, description="Filter by source name"),
    status: Optional[str] = Query(None, description="Filter by status (SUCCESS, ERROR)")
):
    """Get extraction logs"""
    try:
        import sqlite3
        conn = sqlite3.connect(extraction_api.db.db_path)
        cursor = conn.cursor()
        
        sql = """
            SELECT source_name, status, message, timestamp
            FROM extraction_logs
        """
        params = []
        conditions = []
        
        if source:
            conditions.append("source_name = ?")
            params.append(source)
        
        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        logs = []
        
        for row in cursor.fetchall():
            logs.append({
                "source_name": row[0],
                "status": row[1],
                "message": row[2],
                "timestamp": row[3]
            })
        
        conn.close()
        return {"logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting logs: {str(e)}")


# WebSocket endpoint for real-time updates (optional)
@app.websocket("/ws/extraction-status")
async def websocket_extraction_status(websocket):
    """WebSocket endpoint for real-time extraction status updates"""
    await websocket.accept()
    try:
        while True:
            stats = extraction_api.get_extraction_stats()
            await websocket.send_json(stats)
            await asyncio.sleep(30)  # Send updates every 30 seconds
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Data Extraction API Server...")
    print("Available endpoints:")
    print("- GET  /stats - Get extraction statistics")
    print("- POST /extract/run - Run extraction cycle (background)")
    print("- POST /search - Search extracted data")
    print("- GET  /sources - Get data sources")
    print("- POST /export - Export data")
    print("- GET  /recent - Get recent extractions")
    print("- GET  /logs - Get extraction logs")
    
    uvicorn.run(
        "data_extraction_server:app",
        host="127.0.0.1",
        port=6000,
        reload=True,
        log_level="info"
    )
