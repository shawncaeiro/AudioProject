from flask import Flask, render_template, url_for, request, send_from_directory, redirect
from werkzeug import secure_filename
from sidefunctions import getsonglength, combineandsplitsongs, splitty, combinesongs
import logging
import os
# from flask_sslify import SSLify

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['wav'])

application = Flask(__name__)
# application.debug = False
# if 'DYNO' in os.environ: # only trigger SSLify if the app is running on Heroku
#     sslify = SSLify(application)

application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@application.route('/', methods=['GET', 'POST'])
def index():
    application.logger.warning("INDEX!")
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('thanks',
                                    filename=filename, _scheme='https', _external=True))
    return render_template('index.html')

@application.route('/uploadtalk/<filename>', methods=['POST'])
def uploadtalk(filename):
    application.logger.warning("uploadtalk!")
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # application.logger.warning(file.filename)
            # filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], 'voicey' + filename))
            combinedpath = os.path.join(application.config['UPLOAD_FOLDER'], 'combined' + filename)
            split1filename = 'split1' + filename
            split2filename = 'split2' + filename
            split1path = os.path.join(application.config['UPLOAD_FOLDER'], split1filename)
            split2path = os.path.join(application.config['UPLOAD_FOLDER'], split2filename)
            part1len = combineandsplitsongs(os.path.join(application.config['UPLOAD_FOLDER'], filename), os.path.join(application.config['UPLOAD_FOLDER'], 'voicey' + filename), combinedpath, split1path, split2path)
            application.logger.warning(part1len)
    return url_for('combined', filename = filename)

@application.route('/combined/<filename>')
def combined(filename):
    combinedfilename = 'combined' + filename
    combinedpath = os.path.join(application.config['UPLOAD_FOLDER'], 'combined' + filename)
    # split1filename = 'split1' + filename
    # split2filename = 'split2' + filename
    # split1path = os.path.join(application.config['UPLOAD_FOLDER'], split1filename)
    # split2path = os.path.join(application.config['UPLOAD_FOLDER'], split2filename)
    # part1len = combineandsplitsongs(os.path.join(application.config['UPLOAD_FOLDER'], filename), os.path.join(application.config['UPLOAD_FOLDER'], 'voicey' + filename), combinedpath, split1path, split2path)
    # appplication.logger.warning("COMINEDNAME" + combinedfilename)
    # split1filename = 'split1' + filename
    # split2filename = 'split2' + filename
    # splitty(os.path.join(application.config['UPLOAD_FOLDER'], combinedfilename), os.path.join(application.config['UPLOAD_FOLDER'], split1filename), os.path.join(application.config['UPLOAD_FOLDER'], split2filename))
    context = {'combinedfilename':combinedfilename, 'filename':filename}
    return render_template('combined.html', context = context)

@application.route('/split/<filename>')
def split(filename):
    split1filename = 'split1' + filename
    split2filename = 'split2' + filename
    context={'split1name': split1filename, 'split2name':split2filename}
    # combinedpath = os.path.join(application.config['UPLOAD_FOLDER'], 'combined' + filename)
    # split1path = os.path.join(application.config['UPLOAD_FOLDER'], split1filename)
    # split2path = os.path.join(application.config['UPLOAD_FOLDER'], split2filename)
    # part1len = combineandsplitsongs(os.path.join(application.config['UPLOAD_FOLDER'], filename), os.path.join(application.config['UPLOAD_FOLDER'], 'voicey' + filename), combinedpath, split1path, split2path)
    # splitty(os.path.join(application.config['UPLOAD_FOLDER'], filename), split1path, split2path)
    # appplication.logger.warning("COMINEDNAME" + combinedfilename)
    return render_template('split.html', context=context)

@application.route('/thanks/<filename>')
def thanks(filename):
    filepath = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    context = {'filename': filename, 'filepath': filepath}
    # save_spectrogram(filepath, filename)
    return render_template('thanks.html', context = context)

@application.route('/talkover/<filename>')
def talkover(filename):
    uploadurl = '/uploadtalk/' + filename
    truelength = getsonglength(os.path.join(application.config['UPLOAD_FOLDER'], filename))
    length = int(round(truelength))
    lengthms = int(round(truelength * 1000))
    application.logger.warning(length)
    context = {'length':length, 'lengthms':lengthms, 'uploadurl':uploadurl}
    return render_template('talkover.html', context = context)

@application.route('/useruploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(application.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5020))
    application.run(host='0.0.0.0', port=port, debug=True)
    # application.run(port=port, debug=True)
