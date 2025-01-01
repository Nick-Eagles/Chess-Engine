import input_handling
import demonstration

from pyhere import here
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    dirname = input("Load network with which name? ")
    net = load_model(here('nets', dirname, 'model.keras'))

    messDef = 'Which of the following would you like to do:\n'
    options = [
        'Show a sample game for the current network',
        'Play a game against the network'
    ]
    for i, opt in enumerate(options):
        messDef += '(' + str(i+1) + ') ' + opt + '\n'
    messDef += 'Enter 0 to exit: '
    messOnErr = "Not a valid choice"

    choice = 1
    while choice > 0 and choice <= len(options):
        net.summary()
        
        cond = 'var >= 0 and var <= ' + str(len(options))
        choice = input_handling.getUserInput(messDef, messOnErr, 'int', cond)
                                             
        print()

        if choice == 1:
            p = input_handling.readConfig()
            print("Generating the current network's 'best' game...")
            demonstration.bestGame(net)
        elif choice == 2:
            demonstration.interact(net)
