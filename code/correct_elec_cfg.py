import csv

class ElectronConfigurations:
    def __init__(self, filename='../data/External/electronic_configuration.csv'):
        self.configs = {}
        self.fieldnames = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '2p', '3p', '4p', '5p', '6p', '3d', '4d', '5d', '6d', '4f', '5f']
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                self.configs[row['Element']] = {key: int(row[key]) if row[key] else 0 for key in self.fieldnames}

    def query(self, element):
        if element in self.configs:
            config = self.configs[element]
            return [config[key] for key in self.fieldnames]
        else:
            return "Element not found."

if __name__ == '__main__':
    configs = ElectronConfigurations(filename = '../data/External/electronic_configuration.csv')
    print(configs.query('Zn'))
