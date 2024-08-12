from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'habbd6ag63y628##4#t6&&*89*^%$fGbVGFFnN%5$'

app.config['UPLOAD_FOLDER'] = r'base\static\upload'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

from base.com import controller