import time
import cgi
import json
import urllib.parse
import http.server
import os
from player import Player


HOST_NAME = '0.0.0.0'
PORT_NUMBER = 'PORT' in os.environ and int(os.environ['PORT']) or 9000


class PlayerService(http.server.BaseHTTPRequestHandler):

    def do_POST(self):

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        ctype, pdict = cgi.parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = cgi.parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            qs = self.rfile.read(length).decode()
            postvars = urllib.parse.parse_qs(qs, keep_blank_values=1)
        else:
            postvars = {}

        action = postvars['action'][0]

        if 'game_state' in postvars:
            game_state = json.loads(postvars['game_state'][0])
        else:
            game_state = {}


        response = b''
        if action == 'bet_request':
            response = b'%d' % Player().betRequest(game_state)
        elif action == 'showdown':
            Player().showdown(game_state)
        elif action == 'version':
            response = Player.VERSION.encode()

        self.wfile.write(response)

if __name__ == '__main__':
    server_class = http.server.HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), PlayerService)
    print((time.asctime(), "Server Starts - %s:%s" % (HOST_NAME, PORT_NUMBER)))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print((time.asctime(), "Server Stops - %s:%s" % (HOST_NAME, PORT_NUMBER)))
