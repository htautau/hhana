#!/bin/bash

find -H data results -type d | xargs chmod -f 770
find -H data results -type f | xargs chmod -f 660
