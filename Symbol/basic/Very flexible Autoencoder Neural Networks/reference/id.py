print("id(object)는 객체를 입력받아 객체의 고유 주소값(레퍼런스)을 리턴하는 함수입니다.\n")

array1=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
array2=[[1,2,3,4],[5,6,7,8],[9,10,11,12]]
save = [[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]]


print("<계산 전 array1의 id 값> : {}".format(id(array1)))
print("<계산 전 array2의 id 값> : {}\n".format(id(array2)))
print("<계산 전 array1[0]의 id 값> : {}".format(id(array1[0])))
print("<계산 전 array2[0]의 id 값> : {}\n".format(id(array2[0])))
print("<계산 전 save[0]의 id 값> : {}".format(id(save[0])))
print("<계산 전 save[1]의 id 값> : {}".format(id(save[1])))
print("<계산 전 save[2]의 id 값> : {}\n".format(id(save[2])))
print("##################################################")
print("[:]를 사용하지 않고 array1[0]=sava[i]연산을 실행한 경우 array1[0] id는 save[i] id로 바뀝니다.")
for i in range(len(array1)):
    array1[0]=save[i]
    print("save[{}]의 id로 바뀝니다. -> {}".format(i,id(array1[0])))

print("\n")
print("##################################################")
print("[:]를 사용하고, array1[0]=sava[i]연산을 실행한 경우 array2[0] id는 초기 자신의 id를 유지합니다.")
for i in range(len(array2)):
    array2[0][:]=save[i]
    print("array2[0]의 id을 유지합니다 -> {}".format(id(array2[0])))

print("\n")
print("##################################################")
print("전체 list인 array1과 array2의 id는 유지됩니다.")
print("<계산 후 array1의 id 값> : {}".format(id(array1)))
print("<계산 후 array2의 id 값> : {}".format(id(array2)))

