from sys import argv
from xml.dom import minidom
import csv

stem = argv[1][:-4] if argv[1].endswith('.xml') else argv[1]

xmldoc = minidom.parse('%s.xml'%stem)
labellist = xmldoc.getElementsByTagName('label')
labels = [l.attributes['name'].value for l in labellist]
labelset = set(labels)

for split in 'train','test':
    with open('%s-%s.csv'%(stem,split), 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        features = [f for f in reader.fieldnames if f not in labelset]
        x = open('%s-%s.x.txt'%(stem,split), 'w')
        y = open('%s-%s.y.txt'%(stem,split), 'w')
        for row in reader:
            xbuf = ' '.join([row[f] for f in features])
            ybuf = ' '.join([row[l] for l in labels])
            x.write('%s\n'%xbuf)
            y.write("%s\n"%ybuf)
        x.close()
        y.close()
