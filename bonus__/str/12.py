s = input()
str = ''
for i in range(len(s)):
    if i % 3 != 0:
        str += s[i]
print(str)