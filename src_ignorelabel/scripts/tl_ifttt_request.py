 
import requests
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--url', dest='url', default='No image', help='url to the image')

args = parser.parse_args()

def email_alert(first, second, third):
    report = {}
    report["value1"] = first
    report["value2"] = second
    report["value3"] = third
    requests.post("https://maker.ifttt.com/trigger/Fb_stuff_happened/with/key/e5jt9sE0M6cZyLPONDpt7XSoDoQ0BJP-qsrHwWpncyE", data=report)    
    #requests.post("https://maker.ifttt.com/trigger/Tl_stuff_happened/with/key/e5jt9sE0M6cZyLPONDpt7XSoDoQ0BJP-qsrHwWpncyE", data=report)    
    #requests.post("https://maker.ifttt.com/use/lBfXiks2EVc95Lo9iNsG-lNuM76KzCuR1RnEcBf8GWa", data=report)
a=args.url
b='b'
c='c'
email_alert(a, b, c)
