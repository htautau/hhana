#!/bin/bash

find -H data results -type d | xargs chmod 770
find -H data results -type f | xargs chmod 660
