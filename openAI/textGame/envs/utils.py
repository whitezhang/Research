import sys
import six

color2num = dict(
	red = 31,
)

def colorize(string, color, bold=False, highlight = False):
	attr = []
	num = color2num[color]
	if highlight: num += 10
	attr.append(six.u(str(num)))
	if bold: attr.append(six.u('1'))
	attrs = six.u(';').join(attr)
	return six.u('\x1b[%sm%s\x1b[0m') % (attrs, string)
