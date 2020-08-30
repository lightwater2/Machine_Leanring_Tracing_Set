from keras.models import load_model
from datetime import datetime
import logging as log
import shutil


def TimeChecker():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def LoadToModel(callback=None, fileName="null"):
    if fileName == "null":
        return callback()
    return load_model(fileName)


def SaveToModel(model, directory='', time="timeless"):
    print("FileName :" + time)
    model.save(directory + time + '.h5')


def SaveToPredict(pred, callback=None,  directory='', time="timeless"):
    callback(pred).to_csv(directory + time + '.csv', index=False, header=True)


def LoggingHistory(history, time="timeless"):
    log.basicConfig(filename='./archive/' + time + '.txt', level=log.DEBUG,
                    format='# %(message)s')
    log.info('Log Part')
    log.info('loss')
    log.info(history.history['loss'])
    log.info('acc')
    log.info(history.history['accuracy'])
    log.info('val_loss')
    log.info(history.history['val_loss'])
    log.info('val_acc')
    log.info(history.history['val_accuracy'])


def SaveToCode(location, time="timeless"):
    shutil.copy('train.py', location + time + '.txt')