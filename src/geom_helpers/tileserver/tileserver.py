import http.server
import socketserver
from typing import Final

PORT: Final[int] = 8080


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    #def do_GET(self) -> None:
    #    return super().do_GET()


Handler = CORSRequestHandler


with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()

'''
CORS <=> Tileserver CORS Fehler über Maputnik; möglicherweise zu retten über OPTIONS
siehe hier: https://stackoverflow.com/questions/50065875/how-to-enable-cors-in-python
und Background: https://dev.to/ninahwang/cors-explained-enable-in-python-projects-1i96

Es gab noch ein anderes Tutorial für einen Python Webserver 
-> vllt ist dann einfacher; dort wurde die Response explizit programmiert.

def application(environ, start_response):
  if environ['REQUEST_METHOD'] == 'OPTIONS':
    start_response(
      '200 OK',
      [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
        ('Access-Control-Allow-Methods', 'POST'),
      ]
    )
    return ''

'''