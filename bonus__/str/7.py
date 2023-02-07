s=input()
x=s.find('h')
y=s.rfind('h')
print(s[:x]+s[y+1:])