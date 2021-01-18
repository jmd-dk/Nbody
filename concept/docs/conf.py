# Project
project   = 'CONCEPT'
author    = 'Jeppe Dakin'
copyright = '2020, Jeppe Dakin'

# Paths
html_static_path = ['_static']
html_css_files = ['custom.css']
exclude_patterns = ['_build']

# Extensions
extensions = [
    'sphinx_copybutton',
    'sphinx_rtd_theme',
    'sphinx_tabs.tabs',
]

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    # General
    'canonical_url'             : 'https://jmd-dk.github.io/concept/',
    'display_version'           : False,
    'logo_only'                 : False,
    'prev_next_buttons_location': 'both',
    'style_external_links'      : True,
    # Navigation
    'collapse_navigation'        : False,
    'navigation_depth'           : -1,
    'sticky_navigation'          : True,
    'titles_only'                : False,
}

# Syntax highlighting of code blocks
import pygments.styles, pygments.token
def monkeypatch_pygments(name, base_name='default', attrs={}):
    import importlib, sys
    base_module = importlib.import_module('.'.join(['pygments', 'styles', base_name]))
    def name_to_class_name(name):
        return name.capitalize() + 'Style'
    base_class = getattr(base_module, name_to_class_name(base_name))
    styles = getattr(base_class, 'styles', {}).copy()
    styles.update(attrs.pop('styles', {}))
    attrs['styles'] = styles
    class_name = name_to_class_name(name)
    Style = type(class_name, (base_class,), attrs)
    module = type(base_module)(name)
    setattr(module, class_name, Style)
    setattr(pygments.styles, name, module)
    pygments.styles.STYLE_MAP[name] = f'{name}::{class_name}'
    sys.modules['.'.join(['pygments', 'styles', name])] = module
pygments_style = 'concept'
monkeypatch_pygments(
    pygments_style,
    'friendly',
    {
        'background_color': '#f6f6f6',
        'styles': {
            pygments.token.Comment:       'italic #688F98',
            pygments.token.Name.Variable: '#d27a0a',
        },
    },
)

# HTML
html_context           = {'display_github': True}
html_favicon           = html_static_path[0] + '/favicon.ico'
html_last_updated_fmt  = None
html_logo              = html_static_path[0] + '/logo.png'
html_scaled_image_link = False
html_show_copyright    = False
html_show_sourcelink   = False
html_show_sphinx       = False
html_title             = 'CONCEPT Documentation'

