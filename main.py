import pymysql

class Woker:

    def __init__(self):
        try:
            self.conn = pymysql.connect(host='127.0.0.1', port=3306, user='root', password='root', db='mytestdb', charset='utf8')
            self.cur = self.conn.cursor()
            sql1 = '''
                create table if not exists tb_emp(
                    eid int(6) primary key auto_increment,  
                    name varchar(10),
                    sex varchar(5),
                    birthday date,
                    intro varchar(150),
                    profession varchar(10),
                    dept varchar(10),
                    constraint constraint_name_dept foreign key(dept) references tb_dept(name),
                    constraint constraint_name_profession foreign key(profession) references tb_profession(name)
                    );'''
            sql2 = ''' create table if not exists tb_profession(
                    id int(6),
                    name varchar(10) primary key);'''
            sql3 = '''create table if not exists tb_dept(
                    id int(6),
                    name varchar(10) primary key);'''
            self.cur.execute(sql3)
            self.cur.execute(sql2)
            self.cur.execute(sql1)

            self.conn.commit()
        except:
            print("建表失败!")
        else:
            print("建表成功!")

    def add_record(self, sql, param):
        try:
            self.cur.execute(sql)
            self.conn.commit()
        except Exception as e:
            raise e

    def select_record(self, sql, param):
        try:
            cur1 = self.cur.execute(sql)
            return self.cur
        except Exception as e:
            raise e

    def update_record(self, sql, param):
        try:
            self.cur.execute(sql, param)
            self.conn.commit()
        except Exception as e:
            raise e

    def delete_record(self, sql, param):
        try:
            self.cur.execute(sql, param)
            self.conn.commit()
        except Exception as e:
            raise e

    def __del__(self):
        self.cur.close()
        self.conn.close()

if __name__ == '__main__':
    worker = Woker()
    try:
        work_list = [(1, 'Jill', '男', '2000-12-30', 'A good man', '管理','销售'),
                     (2, 'Amy', '女', '1999-9-8', 'A good woman', '管理', '前台')]
        addsql1 = '''replace into tb_emp(eid, name, sex, birthday, intro, profession, dept) values(1, 'Jill', '男', '2000-12-30', 'A good man', '管理','销售')'''
        addsql2 = '''replace into tb_emp(eid, name, sex, birthday, intro, profession, dept) values(2, 'Amy', '女', '1999-9-8', 'A good woman', '管理', '前台')'''
        worker.add_record(addsql1, None)
        worker.add_record(addsql2, None)
        cur = worker.select_record('select * from tb_emp', 'None')
        for row in cur.fetchall():
            print(row)
    except Exception as e:
        raise e

    del worker
