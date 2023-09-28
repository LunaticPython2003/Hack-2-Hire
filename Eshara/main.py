from flask import Flask, Response

app = Flask(__name__)
@app.route('/')
def test():
    return "Hello World"

if __name__=="__main__":
    app.run()