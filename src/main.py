from core import core
from packages.text_predict import text_predict

predictor = text_predict(load_model=False)
prediction = predictor.predict('I need to go to the')
print(prediction)

assistant = core()
assistant.run()
