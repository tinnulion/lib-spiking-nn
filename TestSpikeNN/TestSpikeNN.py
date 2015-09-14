import sys
import time
import numpy
import ctypes
import random
from PIL import Image
import tkinter
from PIL import ImageTk as itk

# Parameters.
DLL_FILE_PATH = r'<your-path-to>\libSpikingNN.dll'
TEMP_DIR = r'<any-temporary-folder-to-place-images>'

GROUP_SIZE = 256
GROUP_NUMBER = 512
NUMBER_OF_INTER_GROUP_CONNECTIONS = 500000
IN_GROUP_WIRE_PROBABILITY = 0.10
DEVICE_ID = 0

# Random input or not?
IS_RANDOM_INPUT = True

################################################################

def get_input_group():
    result = numpy.zeros(GROUP_SIZE, dtype='float32')
    if IS_RANDOM_INPUT:
        for i in range(GROUP_SIZE):
            result[i] = random.choice([0.0, 1.0])
    else:
        result[0] = 1.0 # Only activate first neuron
    return result

def visualize_state(state):
    im = Image.new('RGB', (GROUP_NUMBER, GROUP_SIZE))
    pixels = im.load()
    for i in range(state.shape[0]):
        if state[i] > 0.5:
            x = i // GROUP_SIZE
            y = i % GROUP_SIZE
            pixels[x, y] = (0, 255, 0)
    return im

if __name__ == '__main__':
    print('Starting SpikingNN test...')
    print('Press Ctrl+C when demo bore you.')

    root = tkinter.Tk()
    root.title('Demo of Spiking Network by Igor Ryabtsov')
    root.geometry('%dx%d' % (GROUP_NUMBER, GROUP_SIZE))

    lib = ctypes.windll.LoadLibrary(DLL_FILE_PATH)
    
    lib.Create.restype = ctypes.c_void_p
    c_group_size = ctypes.c_int32(GROUP_SIZE)
    c_group_number = ctypes.c_int32(GROUP_NUMBER)
    c_number_of_igc = ctypes.c_int32(NUMBER_OF_INTER_GROUP_CONNECTIONS)
    c_group_wire_prob = ctypes.c_float(IN_GROUP_WIRE_PROBABILITY)
    c_device_id = ctypes.c_int32(DEVICE_ID)

    handle = lib.Create(
        c_group_size, 
        c_group_number,
        c_number_of_igc,
        c_group_wire_prob,
        c_device_id)

    currentFrame = None

    try:
        epoch = 0
        while True:
            startPoint = time.clock()

            input_group = get_input_group()
            state = numpy.zeros(GROUP_SIZE * GROUP_NUMBER, dtype='float32')

            # Call DLL.
            lib.Iterate.restype = None
            lib.Iterate(handle, input_group.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

            # Visualize result.
            state_visualization = visualize_state(state)

            # Save.
            image_file = TEMP_DIR + '\epoch_' + str(epoch) + '.png'
            state_visualization.save(image_file)

            # Show.
            root.title('Demo of Spiking Network by Igor Ryabtsov -> epoch : ' + str(epoch))
            if currentFrame != None:
                currentFrame.destroy()
            photoImage = tkinter.PhotoImage(file=image_file)
            currentFrame = tkinter.Label(image=photoImage)
            currentFrame.image = photoImage
            currentFrame.pack()
            root.update()

            elapsed = 100 * (time.clock() - startPoint)
            print('Done epoch', epoch, 'in', elapsed, 'millisec.')
            epoch += 1
       
    except KeyboardInterrupt:
        lib.Destroy.restype = None
        lib.Destroy(handle)
        print("Stopped.")
        sys.exit(0)



