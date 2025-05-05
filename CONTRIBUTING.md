# Contributing

Thank you for considering contributing to FuzzyAI! We welcome contributions to improve this project. For general contributions and community guidelines, please see the [repo](https://github.com/cyberark/community/).
## General Steps for Contributing

1. Fork the project.  
2. Clone your fork.  
3. Make local changes to your fork by editing files.  
4. Commit your changes.  
5. Push your local changes to the remote server.  
6. Create a new Pull Request.  

From here, your pull request will be reviewed, and once you've responded to all feedback, it will be merged into the project. Congratulations, you're a contributor!

## Development

To start developing and testing using our development scripts, the following tools need to be installed:

- **FuzzyAI**

### Running Tests

1. Commit and push your changes to your repository.  
2. Clone the repository and run the project with the following commands:  
   ```bash
   cd /tmp
   git clone <your repo>
   cd FuzzyAI//
   poetry install -e .
   fuzzyai fuzz -h
   ```

## Releases

**Maintainers only** should create releases. Follow these steps to prepare for a release:

### Pre-requisites

- Review recent commits and ensure the changelog includes all relevant changes, with references to GitHub issues or PRs when applicable.  
- Verify that any updated dependencies are accurately reflected in the `NOTICES`.  
- Confirm that the required documentation is complete and has been approved.  
