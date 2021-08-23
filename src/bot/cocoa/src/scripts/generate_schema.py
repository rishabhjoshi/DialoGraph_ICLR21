#from lxml import html
import requests
import argparse
import json
import re
import os

# 06-14: remove Name (not a useful language indicator)

def scrape(fname):
    # assume each line in file has form: ENGLISH\tSPANISH
    with open(fname, 'r') as fin:
        content = [line.replace('\n','').split('\t')[0] for line in fin.readlines()]
    return content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema-path')
    parser.add_argument('--cache-path', default='data/cache')
    args = parser.parse_args()

    # Names
    # p = ".//*[@id='content']/table/tr/td[2]/table/tbody/tr[position()<last()]/td"
    # xpath = "%s[%d] | %s[%d]" % (p, 2, p, 4)
    # names = scrape('https://www.ssa.gov/oact/babynames/decades/century.html', xpath, args.cache_path)
    # @eahn: above scraping did not work. I pasted 200 names here instead:
    # names = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Elizabeth', 'William', 'Linda', 'David', 'Barbara', 'Richard', 'Susan', 'Joseph', 'Jessica', 'Thomas', 'Margaret', 'Charles', 'Sarah', 'Christopher', 'Karen', 'Daniel', 'Nancy', 'Matthew', 'Betty', 'Anthony', 'Lisa', 'Donald', 'Dorothy', 'Mark', 'Sandra', 'Paul', 'Ashley', 'Steven', 'Kimberly', 'Andrew', 'Donna', 'Kenneth', 'Carol', 'George', 'Michelle', 'Joshua', 'Emily', 'Kevin', 'Amanda', 'Brian', 'Helen', 'Edward', 'Melissa', 'Ronald', 'Deborah', 'Timothy', 'Stephanie', 'Jason', 'Laura', 'Jeffrey', 'Rebecca', 'Ryan', 'Sharon', 'Gary', 'Cynthia', 'Jacob', 'Kathleen', 'Nicholas', 'Amy', 'Eric', 'Shirley', 'Stephen', 'Anna', 'Jonathan', 'Angela', 'Larry', 'Ruth', 'Justin', 'Brenda', 'Scott', 'Pamela', 'Frank', 'Nicole', 'Brandon', 'Katherine', 'Raymond', 'Virginia', 'Gregory', 'Catherine', 'Benjamin', 'Christine', 'Samuel', 'Samantha', 'Patrick', 'Debra', 'Alexander', 'Janet', 'Jack', 'Rachel', 'Dennis', 'Carolyn', 'Jerry', 'Emma', 'Tyler', 'Maria', 'Aaron', 'Heather', 'Henry', 'Diane', 'Douglas', 'Julie', 'Jose', 'Joyce', 'Peter', 'Evelyn', 'Adam', 'Frances', 'Zachary', 'Joan', 'Nathan', 'Christina', 'Walter', 'Kelly', 'Harold', 'Victoria', 'Kyle', 'Lauren', 'Carl', 'Martha', 'Arthur', 'Judith', 'Gerald', 'Cheryl', 'Roger', 'Megan', 'Keith', 'Andrea', 'Jeremy', 'Ann', 'Terry', 'Alice', 'Lawrence', 'Jean', 'Sean', 'Doris', 'Christian', 'Jacqueline', 'Albert', 'Kathryn', 'Joe', 'Hannah', 'Ethan', 'Olivia', 'Austin', 'Gloria', 'Jesse', 'Marie', 'Willie', 'Teresa', 'Billy', 'Sara', 'Bryan', 'Janice', 'Bruce', 'Julia', 'Jordan', 'Grace', 'Ralph', 'Judy', 'Roy', 'Theresa', 'Noah', 'Rose', 'Dylan', 'Beverly', 'Eugene', 'Denise', 'Wayne', 'Marilyn', 'Alan', 'Amber', 'Juan', 'Madison', 'Louis', 'Danielle', 'Russell', 'Brittany', 'Gabriel', 'Diana', 'Randy', 'Abigail', 'Philip', 'Jane', 'Harry', 'Natalie', 'Vincent', 'Lori', 'Bobby', 'Tiffany', 'Johnny', 'Alexis', 'Logan', 'Kayla']
    # names = scrape('data/names.txt')
    # print '%d names' % len(names)

    # Schools
    # schools = scrape('http://doors.stanford.edu/~sr/universities.html', '//body/ol/li/a', args.cache_path)
    # print '%d schools' % len(schools)

    # Majors
    # xpath = "//body/div/table/td/a | //body/div/table/tr/td/a"
    # majors = scrape('http://www.a2zcolleges.com/majors', xpath, args.cache_path)
    # majors = [re.sub(r'/[^/ ]*', '', major) for major in majors]
    majors = scrape('data/majors.txt')
    print ('%d majors' % len(majors))

    # Companies
    # companies = scrape('https://en.wikipedia.org/wiki/List_of_companies_of_the_United_States', ".//*[@id='mw-content-text']/div/ul/li/a", args.cache_path)

    # Location preference
    companies = scrape('data/loc.txt')
    print ('%d companies' % len(companies))

    # Hobbies
    hobbies = scrape('data/hobbies.txt')
    print ('%d hobbies' % len(hobbies))

    # Location preference
    # loc_pref = scrape('data/loc.txt')

    # Time preference
    time_pref = scrape('data/time.txt')

    # style options!! crux of this research!
    # styles = ['en_lex', 'sp_lex']
    # styles = ['en2sp', 'sp2en']
    # styles = ['en2sp', 'sp2en', 'en_lex', 'sp_lex']
    # styles = ['en2sp', 'sp2en', 'en_lex', 'sp_lex', 'en2sp_soc', 'sp2en_soc', 'en_lex_soc', 'sp_lex_soc']
    # styles = ['random']
    styles = ['en_mono', 'sp_mono']
    print ('%d styles' % len(styles))

    # Schema
    schema = {
        'values': {
            # 'name': names,
            # 'style': styles,
            'major': majors,
            'hobby': hobbies,
            'time_pref': time_pref,
            'company': companies
            },
        'attributes': [
            # {"name": "Style", "value_type": "style", "unique": False},
            # {"name": "Name", "value_type": "name", "unique": False},
            {"name": "Major", "value_type": "major", "unique": False},
            {"name": "Hobby", "value_type": "hobby", "unique": False},
            {"name": "Time", "value_type": "time_pref", "unique": False},
            {"name": "Works at", "value_type": "company", "unique": False}
            ],
        'styles': styles
        }
    with open(args.schema_path, 'w') as out:
        #json.dump(schema, out)
        #print >>out, json.dumps(schema, indent=2, separators=(',', ':'))
        print (json.dumps(schema, indent=2, separators=(',', ':')), file=out)
