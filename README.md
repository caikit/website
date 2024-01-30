# The Caikit documentation

This repository contains the assets for building the [Caikit documentation website](https://caikit.github.io/website/). We welcome contributions from the community to improve and expand our documentation.

- [Building the Caikit website locally](#building-the-caikit-website-locally)
- [Contributing to the docs](#contributing-to-the-caikit-documentation)

## Building the Caikit website locally

The Caikit website is built with the [Just the Docs](https://just-the-docs.com/) Jekyll theme.

### Prerequisites
- [Jekyll](https://jekyllrb.com/)
- [Bundler](https://bundler.io/)
- A clone of the [Caikit website repository](https://github.com/caikit/website) on your local machine

### Building and previewing the Caikit website locally

1. From the command line, change your working directory to the root directory of the `website` repository: `cd <path>/<to>/website`.
2. Run `bundle install`.
3. Run `bundle exec jekyll serve` to build the website.
4. Preview the site in a browser at `localhost:4000`.

**NOTE** You can run the site inside a docker container, but need to pass `--host 0.0.0.0` to `jekyll serve` to allow the site to be loaded via an exposed port in the container

```sh
docker run --rm -it -p 4000:4000 \
    -v $PWD:/src \
    -w /src \
    --entrypoint bash \
    ruby \
    -c "bundle install && bundle exec jekyll serve --host 0.0.0.0 --verbose"
```

## Contributing to the Caikit documentation

First, check out our [contributing guide](https://github.com/caikit/community/blob/main/CONTRIBUTING.md) to learn how to contribute to Caikit. The Caikit documentation follows the same general guidelines as the greater Caikit project.

To contribute to this repository, you'll use the Fork and Pull model common in many open source repositories. For details on this process, check out [The GitHub Workflow
Guide](https://github.com/kubernetes/community/blob/master/contributors/guide/github-workflow.md) from Kubernetes.

1. [Fork this repository](https://docs.github.com/en/get-started/quickstart/fork-a-repo).
2. Make any edits or contributions in your local copy of the repository. The doc files for the website are in the `website/docs` directory.
3. [Create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) from your fork to the upstream repository.

After you create your pull request, a [Caikit doc maintainer](https://github.com/orgs/caikit/teams/caikit-doc-maintainers) will review the change and, if necessary, provide actionable feedback in a timely fashion. Work with your reviewer to make any requested modifications to the pull request. Once the request is approved, a Caikit doc maintainer will merge it to add your contribution to the documentation.
