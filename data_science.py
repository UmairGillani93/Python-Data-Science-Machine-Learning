users = [{'id': 0, 'name':'hero'}, {'id': 1, 'name':'dunn'}, {'id': 2, 'name':'sui'},
{'id': 3, 'name':'chi'}, {'id': 4, 'name':'thor'}, {'id': 5, 'name':'clive'}, {'id': 6, 'name':'hicks'},
{'id': 7, 'name':'devin'}, {'id': 8, 'name':'kate'}, {'id': 9, 'name':'klien'}]

# we have created a list of dictionaries assosciated with corresponding 'keys'
# 'ids'
# each id represents a person

friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
(4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# friendships is a list contains tuples as elements
# each tuple represents ids as friends of each other
# e.g (0,1) and (0,2) show that '0th' person is a friend of both '1' and '2'

for user in users:
    user['friends'] = []

for a,b in friendships:
    users[a]['friends'].append(users[b])
    users[b]['friends'].append(users[a])

# this will add friends of 'a' to the friends of 'b'
# and friends of 'b' to the friends of 'a'
# for every tuple pair in friendships

def number_of_friends(user):
    return len(user['friends'])

total_connections = sum(number_of_friends(user) for user in users)
num_of_friends  = len(users)
avg_connections = total_connections/num_of_friends

number_friends_by_id = [(user['id'], number_of_friends(user)) for user in users]
print(number_friends_by_id)
