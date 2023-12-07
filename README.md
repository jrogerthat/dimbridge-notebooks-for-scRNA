# dimbridge-notebooks-for-scRNA

This is a demo for the Dimbridge design concept applied to scRNA analysis. This repository has the code for the backend notebook, the links to both notebooks, and instructions for running the demo from the two notebooks.
Dimbridge design concept and it's application is a project from Vanderbilt, Tufts, Merck, and ID4.

The backend notebook can be found [here](https://tinyurl.com/dimbridgebackend).
The frontend notebook can be found [here](https://observablehq.com/d/07b13a2cfcdf4659).

## Instructions to run the demo:

## 1. Load the data into google drive:

## 2. Start the backend:
* Give google collab notebook permission to access googel drive.
* Make sure the path is right to the demo data file.
* Run all cells. The server will start on the last cell.
* Copy the url provided by ngrok when the sever starts. You will have to paste this in the top cell of the observable notebook with the variable `predicate_host = "THE NEW URL HERE"`
`
## 3. Start the frontend demo:
* Copy the new url from the running server.
* Refresh the cell, the data should load.
* Happy demoing!
