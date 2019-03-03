from firebase import firebase
firebase = firebase.FirebaseApplication('https://ishita-b18d9.firebaseio.com/', None)
new_user = 'Ozgur Vatansever'

result = firebase.post('/users',{'X_FANCY_HEADER': 'VERY FANCY'})
print result
{u'name': u'-Io26123nDHkfybDIGl7'}

result = firebase.post('/users',{'X_FANCY_HEADER': 'VERY FANCY'})
print result == None
True