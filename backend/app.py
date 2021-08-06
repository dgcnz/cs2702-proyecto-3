from flask import Flask, render_template, request, redirect, url_for, Response, flash, send_from_directory
from werkzeug.utils import secure_filename
import os 
import glob
import time
import json
import importlib
import sys
from lib.rtree import RTree
import face_recognition

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__,
            static_url_path='', 
            static_folder='../frontend/static',
            template_folder='../frontend/templates')
app.secret_key = b'dbii/'


def search_manager(search_type, image_name, k):
   rtree = RTree(128, True)
   image_path = 'data/input/' + image_name
   image = face_recognition.load_image_file(image_path)
   features_vector = face_recognition.api.face_encodings(image)[0]

   answer_paths = []
   if search_type == "knn":
      indices = rtree.knn(features_vector, k)
   elif search_type == "range":
      indices = rtree.contained(features_vector, k)
   for i in indices:
      answer_paths.append(rtree.get(i))
   return answer_paths

@app.route('/')
def home():
   return render_template('finder.html')

@app.route('/find', methods = ['POST'])
def find():
   type_of_search = request.form.get("type")
   if type_of_search == "knn":
      k = int(request.form.get("num_knn"))
   elif type_of_search == "range":
      k = float(request.form.get("num_range"))
   else:
      flash('El tipo de búsqueda no fue seleccionado.', 'alert-danger')
      return redirect(url_for('home'))

   if 'image' not in request.files:
      flash('Por favor seleccione un archivo.', 'alert-warning')
      return redirect(url_for('home'))
   
   file = request.files['image']

   if file.filename == '':
      flash('Archivo no seleccionado', 'alert-danger')
      return redirect(url_for('home'))
   
   if file:
      filename = secure_filename(file.filename)

   files = glob.glob('data/input/*')
   for f in files:
      os.remove(f)
   
   file_to_save =  open("data/input/"+str(file.filename), "wb")
   file_to_save.write(file.read())

   time_start = time.time()
   list_of_images = search_manager(type_of_search,filename,k)
   time_end = time.time()
   flash(u'Se encontraron ' + str(len(list_of_images)) + ' imagenes en ' + str(time_end - time_start) + ' segundos.',  'alert-success')

   #images_output = list()
   #for image_name in list_of_images:
      #images_output.append('lib/' + image_name)
      
   return render_template('finder.html', images_output=list_of_images)
   

@app.route('/<path:foldername>/<path:filename>')
def base_static(foldername, filename):
   print(foldername+'/'+filename)
   return send_from_directory('', foldername + '/' + filename)

if __name__ == '__main__':
   app.run()