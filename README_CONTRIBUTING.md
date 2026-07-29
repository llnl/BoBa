## Contributions

BoBa is an open source project. We welcome contributions via pull requests as well as questions, feature requests, or bug reports via issues. Contact guthrey1@llnl.gov with any questions. Contributions must be made under same license at BoBa itself - see the LICENSE file.

If you aren't a BoBa developer at LLNL, you won't have permission to push new branches to the repository. First, you should create a fork. This will create your own copy of the BoBa repository and ensure you can push your changes up to GitHub and create pull requests.

- Create your branches off the most recent main branch.
- Please rebase your work onto the latest main before requesting a PR.
- Title each PR clearly and give it an unambiguous description.
- Please make pull requests as small and cohesive in scope as is sensible. You may need to break up a PR into multiple PRs grouped by idea or capability.
- Review existing issues before opening a new one. Your issue might already be under development or discussed by others. Feel free to add to any outstanding issue/bug.
- Be explicit when opening issues and reporting bugs. What behavior are you expecting? What is your justification or use case for the new feature/enhancement? How can the bug be recreated? What are any environment variables to consider?
- The examples should remain simple - larger capabilities should be placed in a separate repo with BoBa as a dependency.

Please review the PR checklist below when you are ready for review.

## PR Checklist

- Note that in CI, we treat many warnings as errors in order to improve long term code quality. This may lead to compilation working locally but failing in CI. If you'd like to replicate this behavior locally, use `BOBA_CI=1`. However, we cannot guarantee that the behavior will be completely consistent between your local compiler and the compilers used in CI.
- Was there an issue related to this PR? If so, tag the issue in the description
- Are you adding a new example/test?
  - Please use the `pass_or_fail` machinery found in other examples. Main should return `final_check(check)`
  - First check whether an existing test or miniapp already exercises the changed behavior and only needs new `ci.yaml` coverage
  - Your test should be added to the `all` target in the [Makefile](./Makefile)
  - Please create a make target for your test similar to the existing examples
  - Go to `ci.yaml` and add a test target, skipping platforms if needed
  - `skip` entries are YAML lists, and are matched against the active build/backend tags
```yaml
  test_boba_gmres:            <----- exe name, without the "name flags"
    # GMRES                   <----- comment
    test_gmres_tt:            <----- test name for this exe
      inputs: "-s 0"          <----- use these inputs (e.g. './test_boba_gmres_cpu.out -s 0' )
      skip: [hip, hip300]     <----- platforms on which to skip this test

    # CG
    test_cg_tt:               <----- second test to run for this same exe
      inputs: "-s 1"          <----- inputs for second test
      skip: [hip, hip300]
```
