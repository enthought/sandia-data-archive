major = 0
minor = 2
patch = 0
dev = True

version = '{major}.{minor}.{patch}{dev}'.format(
    major=major, minor=minor, patch=patch, dev='.dev0' if dev else ''
)
