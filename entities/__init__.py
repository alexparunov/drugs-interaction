'''
Each class in xml_classes.py describes the nature of XML file.
The hierarchy of modules from XML is following:
 - Document
   - Sentence
     - Entity and/or Pair

Each class has a constructor whose parameters are existing XML element parameters.
While methods are modifying variable XML elements. For example,

<document id="some-id">
    <sentence id='s1' ... />
    <sentence id='s2' ... />
</document>

The following XML class document will be described by module Document who has id
as its constructor parameter and method add_sentence() to append sentence to list of sentences.
And the same structure is applied to all other classes in xml_classes.py file

While module Parser will parse XML file into above given modules
'''

# Used when we write: from entities import *, in order to import modules to other module
__all__ = ["Document", "Entity", "Pair", "Sentence","Parser"]
