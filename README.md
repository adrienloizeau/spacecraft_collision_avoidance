# Spacecraft Collision Avoidance

Research project conducted with LGI lab at CentraleSupélec to model a Partially Observable Markov Decision Process (POMDP) for Automated Spacecraft Collision Avoidance, focused on spatial debris.

## Directories

- **src Folder**: source code folder
- **test Folder**: Unit tests, integration tests
- **.config Folder**: local configuration related to configuration of project files.
- **.build Folder**: This folder should contain all scripts related to build process (PowerShell, Docker compose…).
- **dep Folder**: This is the directory where all dependencies are stored.
- **doc Folder**: documentation folder
- **.vscode Folder**: Config dir for local VS Code IDE
- **.github Folder**: Config dir for Github workflows & linters

## Run Julia notebook

1. Run julia inside your current project dir: `julia --project=.`
2. Use the right bracket to access the Julia pkg manager: `(pomdp_model) pkg >`
3. Add Pluto to Pkgs: `(pomdp_model) pkg > add Pluto`
4. Come back to julia and tell it to use Pluto: `julia> using Pluto`
5. Run Pluto to access notebook: `julia> Pluto.run()`
