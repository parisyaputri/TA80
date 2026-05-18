from app.__init__ import flask_app

import app.routes


if __name__ == '__main__':

    flask_app.run(
        host='127.0.0.1',
        port=8765,
        debug=False,
        use_reloader=False
    )