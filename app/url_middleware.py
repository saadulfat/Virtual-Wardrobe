"""
URL sanitization middleware for FastAPI
Prevents control character issues in static file URLs
"""

import re
import urllib.parse
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class URLSanitizationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to sanitize URLs and prevent control character issues
    """
    
    def __init__(self, app):
        super().__init__(app)
        # Regex to detect control characters in URLs
        self.control_char_pattern = re.compile(r'[\x00-\x1f\x7f-\x9f]')
        
    async def dispatch(self, request: Request, call_next):
        """
        Process request and sanitize URLs if needed
        """
        # Check if this is a static file request
        if request.url.path.startswith('/static/'):
            # Check for control characters in the URL path
            if self.control_char_pattern.search(request.url.path):
                # Log the issue
                print(f"⚠️  Blocked request with control characters: {repr(request.url.path)}")
                
                # Try to sanitize the URL
                sanitized_path = self.sanitize_url_path(request.url.path)
                
                if sanitized_path != request.url.path:
                    # Redirect to sanitized URL
                    from starlette.responses import RedirectResponse
                    return RedirectResponse(url=sanitized_path, status_code=301)
                else:
                    # Return 404 if can't sanitize
                    return JSONResponse(
                        status_code=404,
                        content={"error": "File not found - invalid characters in URL"}
                    )
        
        # Continue with normal request processing
        response = await call_next(request)
        return response
    
    def sanitize_url_path(self, url_path: str) -> str:
        """
        Sanitize URL path by removing control characters
        """
        # Remove control characters
        sanitized = self.control_char_pattern.sub('', url_path)
        
        # URL decode and re-encode properly
        try:
            decoded = urllib.parse.unquote(sanitized)
            # Re-encode properly
            sanitized = urllib.parse.quote(decoded, safe='/')
        except Exception:
            # If decoding fails, just remove control chars
            pass
        
        return sanitized