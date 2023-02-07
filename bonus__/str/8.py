s = input()
b = s[s.find('h'):s.rfind('h') + 1]
s = s[:s.find('h')]  + b[::-1] +  s[s.rfind('h') + 1:]
print(s)