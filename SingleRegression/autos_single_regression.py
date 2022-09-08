import pandas as pd

database = pd.read_csv('autos.csv', encoding = 'ISO-8859-1')
database = database.drop('dateCrawled', axis = 1)
database = database.drop('dateCreated', axis = 1)
database = database.drop('nrOfPictures', axis = 1)
database = database.drop('postalCode', axis = 1)
database = database.drop('lastSeen', axis = 1)

# initial database analisys
database['name'].value_counts()
database = database.drop('name', axis = 1)
database['seller'].value_counts()
database = database.drop('seller', axis = 1)
database['offerType'].value_counts()
database = database.drop('offerType', axis = 1)

