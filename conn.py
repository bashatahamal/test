import mysql.connector

try:
    db = mysql.connector.connect(
        host='localhost',
        user='root',
        passwd='',
        database='collection'
    )
except mysql.connector.errors.ProgrammingError:
    print('Failed connecting to database')
except mysql.connector.errors.InterfaceError:
    print('Connection refused database is offline')

db_cursor = db.cursor()
font_type = 'PDMS'
char_name = 'Bā’'
marker_type = 'Isolated'
marker_name = 'Kasrah'
image_location = './Multiscale'
QS = 'Al-Baqarah 2'
sql_query = "INSERT INTO dataset (font_type, char_name, marker_type, marker_name,\
             image_location, QS, ID) VALUES (%s, %s, %s, %s, %s, %s, NULL)"
sql_values = (font_type, char_name, marker_type,
              marker_name, image_location, QS)
# db_cursor.execute(sql_query, sql_values)
# db.commit()

# sql_query = "SELECT * FROM `dataset` WHERE `font_type`='PDMS' AND marker_type='Begin'"
sql_query = "SELECT * FROM `dataset` WHERE `font_type`='dfs' AND marker_type=Begin AND char_name=Khā’‬ AND marker_name=Fathahtain"
db_cursor.execute(sql_query)
myresult = db_cursor.fetchall()
print(type(myresult))
print(myresult)
# for x in myresult:
#   print(x)
