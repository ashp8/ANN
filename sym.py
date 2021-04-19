file = open('rfile.txt')
for i in file.readlines():
    i = i.replace(',', ' ');i = i.replace(';', ' ');i = i.replace('\n', '');
    word = i.split(' ');word = [i for i in word if (i !="" )];
    print(word);