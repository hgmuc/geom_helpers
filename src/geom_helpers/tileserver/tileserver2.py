import http.server
import socketserver
#import threading
import os
from http import HTTPStatus
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, Final

from geom_helpers.tiles.terrain_tiles import handle_terrain_tile_request
from geom_helpers.tiles.vector_tiles import handle_mvt_request, TilePbf  # generate_tile_index, write_mvt_tiles, 
from geom_helpers.tiles.xyz_tiles import TILES_DIRECTORY


# Use explicit types for global matplotlib objects
fig: Figure
ax: Axes
fig, ax = plt.subplots(figsize=(9,8))

PORT: Final[int] = 8080

def handle_tile_request(path: str) -> tuple[bool, TilePbf | None]:
    # Extracting Z, X, Y from path
    zxy = path.replace(".png", "").replace(".mvt", "").split("/")
    z,x,y = tuple(zxy[-3:])

    if path.endswith(".mvt"):
        # handle_mvt_request returns (bool, TilePbf)
        done, pbf = handle_mvt_request(int(z),int(x),int(y))
        return done, pbf
    elif path.endswith(".png"):
        # Assuming handle_terrain_tile_request returns a bool
        return handle_terrain_tile_request(int(z), int(x), int(y), ax), None
    else:
        return False, None

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    
    def end_headers(self)-> None:
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', '*')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

    def do_OPTIONS(self)-> None:
        self.send_response(200)
        self.end_headers()

class CustomCORSRequestHandler(CORSRequestHandler):
    # Overwrite init to set different content root directory
    # https://stackoverflow.com/questions/39801718/how-to-run-a-http-server-which-serves-a-specific-path
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # directory is a valid keyword argument for SimpleHTTPRequestHandler in 3.10
        super().__init__(*args, directory=TILES_DIRECTORY, **kwargs)

    def do_GET(self):
        """Serve a GET request."""
        #print(self.request)
        #print("req", self.address_string())
        #print("req", self.command)
        #print("req ressource", self.path, self.path.endswith(".mvt"))
        # send_head() ermittelt den Pfad (Ordner oder File) und 
        # liefert None (Ordner) oder den Inhalt der angeforderten Datei im Binärformat zurück
        # Other content types e.g. PNG graphics for elevation data - see example:
        # https://stackoverflow.com/questions/51142617/displaying-dynamic-webpage-from-python-script-using-http-server

        #if self.path.endswith(".mvt") == True:
        #    #self.writefile("<html><head/><body><p>Hello World</p></body></html>")
        #else:  # Base scenario - return a requested file resource

        print(f"req resource: {self.path} | Path: {os.path.join(TILES_DIRECTORY, self.path[1:])}")
        # In http.server, self.path usually starts with '/'
        local_path = os.path.join(TILES_DIRECTORY, self.path.lstrip('/'))        

        #print("req ressource 1", os.path.exists(os.path.join(TILES_DIRECTORY, self.path[1:])))
        #print("req ressource 2", os.path.join(TILES_DIRECTORY, self.path[1:]))
        done: bool = False
        data: TilePbf | None = None

        if not os.path.exists(local_path):
            done, data = handle_tile_request(self.path)

        if done and data is not None:
            self.send_pbf_response(data)
        else:
            # Fallback to standard file serving
            f = self.send_head()
            if f:
                try:
                    # Ein etwaiger Dateiinhalt von f wird in eine offene Antwortdatei kopiert,
                    # die der Server dann mit ergänztem Header an den Client zurückschickt.
                    self.copyfile(f, self.wfile)
                finally:
                    #print("closing f obj")
                    f.close()

    def send_pbf_response(self, pbf: TilePbf) -> None:
        print(f"Sending pbf response for: {self.path} ({len(pbf)} bytes)")
        # Issue of not returning data to client
        # solved with the help of https://pymotw.com/3/http.server/index.html
        self.send_response(HTTPStatus.OK)
        ctype = self.guess_type(self.path)   # in Python Lib C:\Tools\Anaconda3\envs\osmox\Lib\server.py

        #print("ctype", ctype)
        self.send_header("Content-type", ctype or "application/x-protobuf")
        #self.send_header("Content-type", "text/html; charset=utf-8")
        # OTHER CONTENT TYPES: application/json (JSON/GeoJSON), image/png, application/x-protobuf, text/plain (MVT), ... 
        # (full list defined by IANA)
        # URL to test MVT Tile request: https://tile.nextzen.org/tilezen/vector/v1/512/all/12/2054/1363.mvt?api_key=gCZXZglvRQa6sB2z7JzL1w
        self.send_header("Content-Length", str(len(pbf)))
        #self.send_header("Last-Modified",
        #    self.date_time_string(fs.st_mtime))
        self.end_headers()
        
        try:
            self.wfile.write(pbf)
        except Exception as e:
            print(f"Error writing to wfile: {e}")


    ''' CURRENTLY NOT USED 
    def writefile(self, data: str):
        # Issue of not returning data to client
        # solved with the help of https://pymotw.com/3/http.server/index.html
        self.send_response(HTTPStatus.OK)
        ctype = self.guess_type(self.path)   # in Python Lib C:/Tools/Anaconda3/envs/osmox/Lib/server.py

        #print("ctype", ctype)
        self.send_header("Content-type", ctype)
        #self.send_header("Content-type", "text/html; charset=utf-8")
        # OTHER CONTENT TYPES: application/json (JSON/GeoJSON), image/png, application/x-protobuf, text/plain (MVT), ... 
        # (full list defined by IANA)
        # URL to test MVT Tile request: https://tile.nextzen.org/tilezen/vector/v1/512/all/12/2054/1363.mvt?api_key=gCZXZglvRQa6sB2z7JzL1w
        response_msg = data.encode("UTF-8") + "\r\n".encode("UTF-8")
        self.send_header("Content-Length", str(len(response_msg)))
        #self.send_header("Last-Modified",
        #    self.date_time_string(fs.st_mtime))
        self.end_headers()
        
        try:
            self.wfile.write(response_msg)
        except Exception as e:
            print(e, data)
    '''

#class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
#    pass



Handler = CustomCORSRequestHandler
Handler.directory = TILES_DIRECTORY


#generate_tile_index()
#write_mvt_tiles()


if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"serving at port {PORT}")
        os.chdir(TILES_DIRECTORY)
        print("CWD", os.getcwd())
        httpd.serve_forever()



'''
with ThreadedTCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    os.chdir(TILES_DIRECTORY)
    print("CWD", os.getcwd())
    ip, port = httpd.server_address

    # Start a thread with the server -- that thread will then start one
    # more thread for each request
    server_thread = threading.Thread(target=httpd.serve_forever)
    # Exit the server thread when the main thread terminates
    server_thread.daemon = True
    server_thread.start()
    print("Server loop running in thread:", server_thread.name)

    #httpd.serve_forever()
'''


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