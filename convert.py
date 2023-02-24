import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


def impedance_phase_csv():
    for label_number in os.listdir('./raw_data'):
        for mouse in os.listdir('./raw_data/' + label_number):
            for measurement in os.listdir('./raw_data/' + label_number + '/' + mouse):
                data = pd.read_csv('./raw_data/' + label_number + '/' + mouse + '/' + measurement)
                result = pd.DataFrame(columns=('frequency', 'impedance', 'phase'))
                for x in range(11, 51):
                    impedance_real = data.iat[x, 2]
                    impedance_image = data.iat[x, 3]
                    impedance = np.sqrt(impedance_real ** 2 + impedance_image ** 2)
                    phase = np.arctan(impedance_image/impedance_real) * 180/np.pi
                    frequency = data.iat[x, 1]
                    result = result.append(pd.DataFrame({'frequency': [frequency], 'impedance': [impedance], 'phase': [phase]}))

                name1 = measurement.split('.')[0]
                electrodes = name1.split('_')[1]

                folder = os.getcwd() + './impedance_data/' + label_number + '/' + mouse
                if not os.path.exists(folder):
                    os.makedirs(folder)
                result.to_csv(os.getcwd() + './impedance_data/' + label_number + '/' + mouse + '/' + electrodes + '_impedance_phase.csv', sep=",", index=False)

def npy_generate():
    for label_number in os.listdir('./impedance_data'):
        for mouse in os.listdir('./impedance_data' + '/' + label_number):
            frequency = pd.read_csv('./frequency.csv', usecols=['frequency'], nrows=40)
            df = pd.DataFrame(frequency)
            for measurement in os.listdir('./impedance_data' + '/' + label_number + '/' + mouse):
                impedance_data = pd.read_csv('./impedance_data' + '/' + label_number + '/' + mouse + '/' + measurement,  usecols=['impedance'], nrows=40)
                print(measurement.split('.')[0])
                df.insert(df.shape[1], measurement.split('.')[0], impedance_data)
            npy = df.drop(['frequency'], axis=1).to_numpy()
            print(npy.shape)
            folder = os.getcwd() + './npy_data/' + label_number
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save('./npy_data/' + label_number + '/' + mouse + '.npy', npy)


if __name__ == '__main__':
    impedance_phase_csv()
    npy_generate()




