from flask import Flask, render_template, url_for, request
from PIL import Image
import os

from result import diagnosis

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image file from the form
        image_file = request.files['image']
        name = image_file.filename
        image_path = os.path.join(app.static_folder, 'uploads', image_file.filename)
        image_file.save(image_path)
        # Read the image using PIL (Python Imaging Library)
        image = Image.open(image_path)

        diagnos = diagnosis(image_file)
        # Render the result template with the image size
        return render_template('result.html', diagnos=diagnos, image_name=f'/uploads/{name}')

        # Render the upload form template
    return render_template('index.html')


@app.route('/result')
def about():
    return 'Hello World1!'


if __name__ == '__main__':
    app.run(debug=True)
