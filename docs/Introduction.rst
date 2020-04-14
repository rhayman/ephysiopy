Introduction
============

What is ephysiopy?
------------------

A collection of tools for analysing electrophysiology data recorded with either the openephys or Axona recording
systems.

Rationale
---------

Analysis using Axona's Tint cluster cutting program or phy/ phy2 (openephys) is great but limited. This extends that functionality.

Installation
------------

ephysiopy requires python3.6 or greater. The easiest way to install is using pip:

``python3 -m pip install ephysiopy``

or,

``pip3 install ephysiopy``

Or similar. There are dependencies (see requirements.txt) but pip should take care of this. I haven't yet tried this in a virtual environment like conda or similar but a quick google shows it should be pretty easy. Let me know if it isn't.

Documentation
-------------

You're looking at it but for completeness it is at `https://ephysiopy.readthedocs.io/ <https://ephysiopy.readthedocs.io/>`_

Updates etc
-----------

The code is maintained in two locations, github and PyPi. The documentation of the code is held on `readthedocs <https://readthedocs.org>`_ and will automatically update whenever new code is pushed to github. For completeness the github repository is `here <https://github.com/rhayman/ephysiopy>`_ and  PyPi repo is `here <https://pypi.org/project/ephysiopy/>`_.

I tend to push to github more as the structure of the project (package and sub-package locations and so on) probably isn't going to change much from now on, but the documentation could always do with being updated.

If you want to update then it's the usual method with pip:

``pip3 install ephysiopy --upgrade``

or,

``python3 -m pip install ephysiopy --upgrade``

Issues/ development etc
-----------------------

The place to point out problems is at the github repository `https://github.com/rhayman/ephysiopy <https://github.com/rhayman/ephysiopy>`_. Alternatively you can email me.