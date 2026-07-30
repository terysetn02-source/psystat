; psystat_setup.iss
; -----------------
; Inno Setup 6 script for PsyStat Windows installer.
;
; Prerequisites:
;   1. Run PyInstaller first:  pyinstaller psystat.spec
;      This produces dist\PsyStat\ with PsyStat.exe and all bundled files.
;   2. Install Inno Setup 6 (https://jrsoftware.org/isinfo.php)
;   3. Open this file in the Inno Setup IDE and click Build → Compile,
;      OR run from the command line:
;        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" psystat_setup.iss
;
; Output: installer\Output\PsyStat-Setup-1.1.0.exe

#define AppName      "PsyStat"
#define AppVersion   "1.1.0"
#define AppPublisher "Tery Setiawan"
#define AppURL       "https://github.com/terysetn02-source/psystat"
#define AppExeName   "PsyStat.exe"
#define SourceDir    "..\dist\PsyStat"

[Setup]
; Basic metadata
AppId={{A3F7C2D1-8B4E-4F9A-B5C6-1D2E3F4A5B6C}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases

; Installer behaviour
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=PsyStat-Setup-{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern

; Security — allow non-admin install to AppLocalData if UAC fails
PrivilegesRequiredOverridesAllowed=dialog

; Icons
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#AppExeName}

; Windows version requirement (Windows 10 and above)
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main application folder (all PyInstaller output)
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";             Filename: "{app}\{#AppExeName}"
Name: "{group}\Uninstall {#AppName}";   Filename: "{uninstallexe}"
Name: "{commondesktop}\{#AppName}";     Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Show a SmartScreen bypass reminder after install on Windows 10/11
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox(
      'PsyStat has been installed successfully.' + #13#10 + #13#10 +
      'IMPORTANT — First Launch:' + #13#10 +
      'Windows SmartScreen may block the app the first time you run it.' + #13#10 +
      'If a blue warning screen appears, click "More info" then "Run anyway".' + #13#10 + #13#10 +
      'This warning is normal for new software and does not indicate any risk.',
      mbInformation, MB_OK
    );
  end;
end;
