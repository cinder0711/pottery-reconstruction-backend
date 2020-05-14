import os
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, session, send_from_directory, abort
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
from test import startModel, runModel

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')



UPLOAD_FOLDER = './data/pottery'
DOWNLOAD_FOLDER = './data/pottery/send'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'pgm'])

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

model0,model1,model2,model3,model4,model5,model6,model7=startModel()

@app.route('/upload', methods=['POST'])
def fileUpload():
    logger.info("welcome to upload`")
    file = request.files['file'] 
    filename = secure_filename(file.filename)
    target=os.path.join(UPLOAD_FOLDER,filename[:-4])
    if not os.path.isdir(target):
        os.mkdir(target)
        target=os.path.join(target,'test')
        os.mkdir(target)
    destinationUpload=os.path.join(target, filename)
    file.save(destinationUpload)
    session['uploadFilePath']=destinationUpload


    runModel(filename,model0,model1,model2,model3,model4,model5,model6,model7)

    try:
        return send_from_directory(app.config["DOWNLOAD_FOLDER"], filename=filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,host="0.0.0.0",use_reloader=False)
