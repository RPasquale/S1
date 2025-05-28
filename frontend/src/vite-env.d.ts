/// <reference types="vite/client" />
import 'react';

declare module 'react' {
  interface InputHTMLAttributes<T> {
    /** Enable folder selection in file inputs (string to satisfy ReactDOM non-standard attr) */
    webkitdirectory?: boolean | string
  }
}
