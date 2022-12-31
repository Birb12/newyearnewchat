import os
import csv

user_mannerisms = []
f = open('userdataset.csv', 'w', newline='')
writer = csv.writer(f)
header = ['text']
writer.writerow(header)

print("Please type in 10 phrases you normally use to use as a baseline: ")

for i in range (0, 10):
    text = input("Text: ")
    format = [text]

    writer.writerow(format)
