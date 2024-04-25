from Dataset import *
from Neurolink import *
from sklearn.feature_extraction.text import TfidfVectorizer

# input parameters +
PATH = 'beethoven_piano_sonatas\measures\\01-1.measures.tsv'
table_measures = getData(PATH) #list[list[str]]


########## table vectorisation + 
table_text = [' '.join(row) for row in table_measures] #to list[str]
vectorizer = TfidfVectorizer()
vectorised_table = vectorizer.fit_transform(table_text)
dense_matrix = vectorised_table.toarray()
print(vectorised_table)

########## vectorised table to tenzor transformation + 
inputs = torch.tensor(dense_matrix, dtype=torch.float)

######### neurolink parameters +
input_size_measures = inputs.shape[1]  #Number of columns in input table
hidden_size_measures = 20 #Number of neurons
output_size_measures = input_size_measures #Number of columns in output table
num_epochs = 10

########## model connection +
model = SimpleMusicModel(input_size_measures, hidden_size_measures, output_size_measures)
criterion = nn.MSELoss()  # loss fun
optimizer = optim.Adam(model.parameters(), lr=0.001)

########## neurolink study
for epoch in range(num_epochs):
    outputs = model(inputs)
    loss = criterion(outputs, outputs) # (outputs, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


