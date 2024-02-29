# Open function to open the file "MyFile1.txt"  
# (same directory) in read mode and 
str1="pippo \n"
str2 = input("digita una stringa:\n")

file1 = open("prova.txt", "a+") 
file1.write(str1)
file1.write(str2 + "\n")
file1.write("stop")
file1.close()
    
# store its reference in the variable file1  
# and "MyFile2.txt" in D:\Text in file2 
#file2 = open(r"D:\Text\MyFile2.txt", "w+") 