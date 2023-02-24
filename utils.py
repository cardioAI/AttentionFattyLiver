import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')


class Convert:
    def __init__(self, url):
        self.url = url
    def impedance_phase_csv(self):
        for sample in os.listdir(self.url):
            url1 = self.url + '/' + sample
            for measurement in os.listdir(url1):
                url2 = url1 + '/' + measurement
                data = pd.read_csv(url2)
                result = pd.DataFrame(columns=('frequency', 'impedance', 'phase'))
                for x in range(11, 51):
                    impedance_real = data.iat[x, 2]
                    impedance_image = data.iat[x, 3]
                    impedance = np.sqrt(impedance_real ** 2 + impedance_image ** 2)
                    phase = np.arctan(impedance_image / impedance_real) * 180 / np.pi
                    frequency = data.iat[x, 1]
                    result = result.append(pd.DataFrame({'frequency': [frequency], 'impedance': [impedance], 'phase': [phase]}))

                    name1 = measurement.split('.')[0]
                    electrodes = name1.split('_')[1]

                    folder = './predict_impedance' + '/' + sample
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    result.to_csv(folder + '/' + measurement + '_impedance_phase.csv', sep=",", index=False)

    def npy_generate(self):
        url1 = '/predict_impedance'
        for sample in os.listdir(url1):
            url2 = url1 + '/' + sample
            frequency = pd.read_csv('./frequency.csv', usecols=['frequency'], nrows=40)
            df = pd.DataFrame(frequency)
            for measurement in os.listdir(url2):
                impedance_data = pd.read_csv(url2 + '/' + measurement, usecols=['impedance'], nrows=40)
                df.insert(df.shape[1], measurement.split('.')[0], impedance_data)
            npy = df.drop(['frequency'], axis=1).to_numpy()
            folder = './predict_results'
            if not os.path.exists(folder):
                os.makedirs(folder)
            np.save(folder + '/' + sample + '.npy', npy)

# if __name__ == '__main__':
    # cnpy = Convert('./predict_data')
    # cnpy.impedance_phase_csv()
    # cnpy.npy_generate()