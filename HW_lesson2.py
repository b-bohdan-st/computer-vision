import cv2

image = cv2.imread("photo.png")
resized = cv2.resize(image, (400, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(image, 200, 200) #200, 200 - не оптимізовано
cv2.imshow("Photo",image)
cv2.imshow("Resized", resized)
cv2.imshow("Gray", gray)
cv2.imshow("Edges", edges)

#email_photo
email = cv2.imread("email.png")
resized_email = cv2.resize(email, (600, 200))
gray_email = cv2.cvtColor(email, cv2.COLOR_BGR2GRAY)
edges_email = cv2.Canny(email, 100, 100)  #100, 100 - оптимізовано

cv2.imshow("Email", email)
cv2.imshow("Resized_email", resized_email)
cv2.imshow("Gray_email", gray_email)
cv2.imshow("Edges_email", edges_email)

cv2.waitKey(0)
cv2.destroyAllWindows()
