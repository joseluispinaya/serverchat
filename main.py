# This is a sample Python script.
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import load_resources, predict_class, get_response
import logging
import os

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar recursos al iniciar la aplicación
# intents, words, classes, model = load_resources()

# Cargar recursos al iniciar la aplicación
try:
    logging.info("Cargando recursos...")
    intents, words, classes, model = load_resources()
    logging.info("Recursos cargados correctamente.")
except Exception as e:
    logging.error(f"Error al cargar recursos: {e}")
    raise  # Detener la aplicación si no se pueden cargar los recursos


@app.route("/")
def home():
    return "¡El servidor está funcionando correctamente!"

@app.route('/api/get_response', methods=['POST'])
def chatbot_response():
    """
    Procesa un mensaje del usuario y devuelve una respuesta del chatbot.

    Parámetros (en el cuerpo de la solicitud JSON):
    - message (str): El mensaje del usuario.

    Respuesta (JSON):
    - response (str): La respuesta del chatbot.
    - error (str, opcional): Mensaje de error en caso de fallo.
    """
    try:
        user_message = request.json.get('message', '').strip()
        if not user_message:
            return jsonify({'error': 'El mensaje no puede estar vacío.'}), 400

        tag = predict_class(user_message, words, classes, model)
        #print(f"Tag es: {tag}")
        #logging.info(f"Tag predicho: {tag}")

        response = get_response(tag, intents)
        return jsonify({'response': response})

    except Exception as e:
        logging.error(f"Error en chatbot_response: {e}")
        return jsonify({'error': 'Hubo un error al procesar tu solicitud.'}), 500


if __name__ == "__main__":
    # Usar el puerto que Render asigna o 5000 si estás localmente
    port = int(os.environ.get('PORT', 5000))
    logging.info(f"Iniciando servidor en el puerto {port}...")
    app.run(host='0.0.0.0', port=port)
