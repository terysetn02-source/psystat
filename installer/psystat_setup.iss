; Inno Setup script for the PsyStat Windows installer.
; Build (after `pyinstaller --noconfirm psystat.spec` has created dist\PsyStat):
;     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" /DMyAppVersion=1.2.0 installer\psystat_setup.iss
; Output: installer\Output\PsyStat-Setup-<version>.exe

#ifndef MyAppVersion
  #define MyAppVersion "1.2.0"
#endif
#define MyAppName "PsyStat"
#define MyAppPublisher "Tery Setiawan"
#define MyAppURL "https://github.com/terysetn02-source/psystat"
#define MyAppExeName "PsyStat.exe"

[Setup]
; IMPORTANT: AppId identifies this application for upgrades and uninstalling. NEVER change it between versions.
; (If an earlier PsyStat installer used a different AppId, put that GUID here so that new versions upgrade in place.)
AppId={{4CC540F4-59E8-4EE6-959D-1D18B51BCB88}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\PsyStat
DefaultGroupName=PsyStat
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=Output
OutputBaseFilename=PsyStat-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Install for the current user without administrator rights unless the user chooses "for all users"
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#MyAppExeName}
#if FileExists("..\assets\icon.ico")
SetupIconFile=..\assets\icon.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\PsyStat\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\PsyStat"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\PsyStat"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
