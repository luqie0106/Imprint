; *** Inno Setup version 6.1.0+ Chinese Simplified messages ***
;
[LangOptions]
LanguageName=Chinese (Simplified)
LanguageID=$0804
LanguageCodePage=0

[Messages]

; *** Application titles
SetupAppTitle=安装
SetupWindowTitle=安装 - %1
UninstallAppTitle=卸载
UninstallWindowTitle=卸载 - %1

; *** Misc. common
ExitSetupTitle=退出安装程序
ExitSetupMessage=安装尚未完成。如果您此时退出，本程序将不会被安装。%n%n您可以稍后再次运行安装程序以完成安装。%n%n要退出安装程序吗？
AboutSetupMenuItem=关于安装程序(&A)...
AboutSetupTitle=关于安装程序
AboutSetupMessage=%1 版本 %2%n%3%n%n%1 网站:%n  %4
AboutSetupNote=
TranslatorNote=

; *** Buttons
ButtonBack=< 上一步(&B)
ButtonNext=下一步(&N) >
ButtonInstall=安装(&I)
ButtonOK=确定
ButtonCancel=取消
ButtonYes=是(&Y)
ButtonYesToAll=全是(&A)
ButtonNo=否(&N)
ButtonNoToAll=全否(&O)
ButtonFinish=完成(&F)
ButtonBrowse=浏览(&B)...
ButtonWizardBrowse=浏览(&R)...
ButtonNewFolder=新建文件夹(&M)

; *** "Select Language" dialog messages
SelectLanguageTitle=选择安装语言
SelectLanguageLabel=选择安装时使用的语言:

; *** Common wizard text
ClickNext=单击“下一步”继续，或单击“取消”退出安装程序。
BeveledLabel=
BrowseDialogTitle=浏览文件夹
BrowseDialogLabel=在下面的列表中选择一个文件夹，然后单击“确定”。
NewFolderName=新建文件夹

; *** "Welcome" wizard page
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=这将在您的电脑上安装 [name/ver]。%n%n建议您在继续之前关闭所有其他应用程序。

; *** "Password" wizard page
WizardPassword=密码
PasswordLabel1=此安装程序受密码保护。
PasswordLabel3=请输入密码，密码区分大小写，然后单击“下一步”继续。
PasswordEditLabel=密码(&P):
IncorrectPassword=您输入的密码不正确，请重新输入。

; *** "License Agreement" wizard page
WizardLicense=许可协议
LicenseLabel=请在继续之前仔细阅读以下重要信息。
LicenseLabel3=请阅读以下许可协议。在继续安装之前，您必须接受此协议的条款。
LicenseAccepted=我接受协议(&A)
LicenseNotAccepted=我拒绝协议(&D)

; *** "Information" wizard pages
WizardInfoBefore=信息
InfoBeforeLabel=请在继续之前阅读以下重要信息。
InfoBeforeClickLabel=准备好继续安装程序后，请单击“下一步”。
WizardInfoAfter=信息
InfoAfterLabel=请在继续之前阅读以下重要信息。
InfoAfterClickLabel=准备好继续安装程序后，请单击“下一步”。

; *** "User Information" wizard page
WizardUserInfo=用户信息
UserInfoDesc=请输入您的信息。
UserInfoName=用户姓名(&U):
UserInfoOrg=所属组织(&O):
UserInfoSerial=序列号(&S):
UserInfoNameCodes=

; *** "Select Destination Location" wizard page
WizardSelectDir=选择目标位置
SelectDirDesc=您想将 [name] 安装到哪个文件夹？
SelectDirLabel3=安装程序将把 [name] 安装到以下文件夹中。
SelectDirBrowseLabel=要继续，请单击“下一步”。如果您想选择其他文件夹，请单击“浏览”。
DiskSpaceMBLabel=此程序至少需要 [mb] MB 的可用磁盘空间。
CannotInstallToNetworkDrive=安装程序不能安装到网络驱动器。
CannotInstallToUNCPath=安装程序不能安装到 UNC 路径。
InvalidPath=您必须输入一个包含驱动器盘符的完整路径，例如:%n%nC:\APP%n%n或形如以下格式的 UNC 路径:%n%n\\server\share
InvalidDrive=您选择的驱动器或 UNC 共享不存在或无法访问，请选择其他位置。
DiskSpaceWarningTitle=磁盘空间不足
DiskSpaceWarning=安装程序至少需要 %1 KB 的可用空间才能安装，但所选驱动器上只有 %2 KB 可用。%n%n您仍要继续吗？
DirNameTooLong=文件夹名称或路径太长。
InvalidDirName=文件夹名称无效。
BadDirName386=文件夹名称中不能包含以下字符:%n%n%1
DirExistsTitle=文件夹已存在
DirExists=文件夹:%n%n%1%n%n已经存在。您仍要安装到此文件夹中吗？
DirDoesntExistTitle=文件夹不存在
DirDoesntExist=文件夹:%n%n%1%n%n不存在。您要创建此文件夹吗？

; *** "Select Components" wizard page
WizardSelectComponents=选择组件
SelectComponentsDesc=应安装哪些组件？
SelectComponentsLabel2=选择您想要安装的组件；清除您不想安装的组件。准备好后，单击“下一步”。
FullInstallation=完全安装
CompactInstallation=精简安装
CustomInstallation=自定义安装
NoSubComponentsAvailable=无可用子组件。
ComponentSize1=%1 KB
ComponentSize2=%1 MB
ComponentsDiskSpaceMBLabel=当前选择的组件至少需要 [mb] MB 的可用磁盘空间。

; *** "Select Additional Tasks" wizard page
WizardSelectTasks=选择附加任务
SelectTasksDesc=您想要执行哪些附加任务？
SelectTasksLabel2=选择您希望安装程序在安装 [name] 时执行的附加任务，然后单击“下一步”。

; *** "Select Start Menu Folder" wizard page
WizardSelectProgramGroup=选择开始菜单文件夹
SelectStartMenuFolderDesc=安装程序应在哪里放置程序的快捷方式？
SelectStartMenuFolderLabel3=安装程序将在以下开始菜单文件夹中创建程序的快捷方式。
SelectStartMenuFolderBrowseLabel=要继续，请单击“下一步”。如果您想选择其他文件夹，请单击“浏览”。
MustEnterGroupName=您必须输入一个文件夹名称。
GroupNameTooLong=文件夹名称或路径太长。
InvalidGroupName=文件夹名称无效。
BadGroupName=文件夹名称中不能包含以下字符:%n%n%1
NoProgramGroupCheck2=不创建开始菜单文件夹(&D)

; *** "Ready to Install" wizard page
WizardReady=准备安装
ReadyLabel1=安装程序现在已准备好开始在您的电脑上安装 [name]。
ReadyLabel2a=单击“安装”以继续安装，或单击“上一步”以查看或更改任何设置。
ReadyLabel2b=单击“安装”以继续安装。
ReadyMemoUserInfo=用户信息:
ReadyMemoDir=目标位置:
ReadyMemoType=安装类型:
ReadyMemoComponents=选定组件:
ReadyMemoGroup=开始菜单文件夹:
ReadyMemoTasks=附加任务:

; *** TDownloadWizardPage message strings
DownloadingLabel=正在下载附加文件...
ButtonStopDownload=停止下载(&S)
StopDownload=您确定要停止下载吗？
ErrorDownloadAborted=下载已中止
ErrorDownloadFailed=下载失败: %1 %2
ErrorDownloadSizeFailed=获取大小失败: %1 %2
ErrorFileHash1=文件哈希不匹配: %1
ErrorFileHash2=无效的文件哈希: 预期为 %1，实际为 %2
ErrorProgress=无效的进度: %1 之 %2
ErrorFileSize=无效的文件大小: 预期为 %1，实际为 %2

; *** "Preparing to Install" wizard page
WizardPreparing=正在准备安装
PreparingDesc=安装程序正在准备在您的电脑上安装 [name]。
PreviousInstallNotCompleted=以前程序的安装/卸载尚未完成。您需要重新启动电脑以完成该安装。%n%n重新启动电脑后，请再次运行安装程序以完成 [name] 的安装。
CannotContinue=安装程序无法继续。请单击“取消”退出。
ApplicationsFound=以下应用程序正在使用需要由安装程序更新的文件。建议您允许安装程序自动关闭这些应用程序。
ApplicationsFound2=以下应用程序正在使用需要由安装程序更新的文件。建议您允许安装程序自动关闭这些应用程序。安装完成后，安装程序将尝试重新启动这些应用程序。
CloseApplications=自动关闭应用程序(&A)
DontCloseApplications=不关闭应用程序(&D)
ErrorCloseApplications=安装程序无法自动关闭所有应用程序。建议您在继续之前关闭所有使用需要由安装程序更新的文件的应用程序。
PrepareToInstallNeedsRestart=安装程序必须重新启动电脑。重新启动电脑后，请再次运行安装程序以完成 [name] 的安装。%n%n您现在想重新启动吗？

; *** "Installing" wizard page
WizardInstalling=正在安装
InstallingLabel=安装程序正在将 [name] 安装到您的电脑中，请稍候。

; *** "Setup Completed" wizard page
FinishedHeadingLabel=[name] 安装向导完成
FinishedLabelNoIcons=安装程序已在您的电脑上安装了 [name]。
FinishedLabel=安装程序已在您的电脑上安装了 [name]。可以通过选择已安装的图标来启动该应用程序。
ClickFinish=单击“完成”退出安装程序。
FinishedRestartLabel=为了完成 [name] 的安装，安装程序必须重新启动您的电脑。您现在想重新启动吗？
FinishedRestartMessage=为了完成 [name] 的安装，安装程序必须重新启动您的电脑。%n%n您现在想重新启动吗？
ShowReadmeCheck=是的，我想查看自述文件
YesRadio=是，现在重新启动电脑(&Y)
NoRadio=否，我稍后重新启动电脑(&N)
RunEntryExec=运行 %1
RunEntryExecArgs=运行 %1 (%2)
RunEntryShellExec=查看 %1

; *** "Setup Needs the Next Disk" popup msgs
ChangeDiskTitle=安装程序需要下一张磁盘
SelectDiskLabel2=请插入磁盘 %1 并单击“确定”。%n%n如果此磁盘上的文件可以在不同于下面显示的文件夹中找到，请输入正确的路径或单击“浏览”。
PathLabel=路径(&P):
FileNotInDir2=文件“%1”无法在“%2”中找到。请插入正确的磁盘或选择其他文件夹。
SelectDirectoryLabel=请指定下一张磁盘的位置。

; *** Installation status messages
StatusClosingApplications=正在关闭应用程序...
StatusCreateDirs=正在创建目录...
StatusExtractFiles=正在提取文件...
StatusCreateIcons=正在创建快捷方式...
StatusCreateIniEntries=正在创建 INI 条目...
StatusCreateRegistryEntries=正在创建注册表项...
StatusRegisterFiles=正在注册文件...
StatusSavingVersionCache=正在保存版本信息...
StatusSetupRemoved=安装程序已从您的电脑中删除。
StatusRestartingApplications=正在重新启动应用程序...

; *** Misc. errors
FileAbortRetryIgnore=单击“重试”以再次尝试，单击“忽略”以跳过此文件(不推荐)，或单击“中止”以取消安装。
FileAbortRetryIgnore2=单击“重试”以再次尝试，单击“忽略”以继续(不推荐)，或单击“中止”以取消安装。
SourceIsCorrupted=源文件损坏
SourceDoesntExist=源文件“%1”不存在
ExistingFileReadOnly2=无法替换现有文件，因为该文件被标记为只读。
ExistingFileReadOnlyRetry=单击“重试”以删除只读属性并重试，单击“忽略”以跳过此文件，或单击“中止”以取消安装。
ErrorReadingExistingDest=尝试读取现有文件时出错:
FileExistsSelectAction=选择操作
FileExists2=该文件已经存在。
FileExistsOverwriteExisting=覆盖现有文件(&O)
FileExistsKeepExisting=保留现有文件(&K)
FileExistsOverwriteOrKeepAll=对后续冲突执行此操作(&D)
FileExistsKeepAll=全部保留(&A)
FileExistsOverwriteAll=全部覆盖(&V)
ExistingFileNewerSelectAction=选择操作
ExistingFileNewer2=现有文件比安装程序尝试安装的文件更新。建议您保留现有文件。
ExistingFileNewerOverwriteExisting=覆盖现有文件(&O)
ExistingFileNewerKeepExisting=保留现有文件(&K)(推荐)
ExistingFileNewerOverwriteOrKeepAll=对后续冲突执行此操作(&D)
ExistingFileNewerKeepAll=全部保留(&A)(推荐)
ExistingFileNewerOverwriteAll=全部覆盖(&V)
ErrorChangingAttr=尝试更改现有文件的属性时出错:
ErrorCreatingTemp=尝试在目标目录中创建文件时出错:
ErrorReadingSource=尝试读取源文件时出错:
ErrorCopying=尝试复制文件时出错:
ErrorReplacingExistingFile=尝试替换现有文件时出错:
ErrorRestartReplace=重新启动替换失败:
ErrorRenamingTemp=尝试重命名目标目录中的文件时出错:
ErrorRegisterServer=无法注册 DLL/OCX: %1
ErrorRegSvr32Failed=RegSvr32 失败并退出代码 %1
ErrorRegisterTypeLib=无法注册类型库: %1

; *** Post-installation errors
ErrorOpeningReadme=尝试打开自述文件时出错。
ErrorRestartingController=无法重新启动应用程序管理器。您需要手动重新启动电脑。

; *** Uninstaller strings
UninstallNotFound=文件“%1”不存在。无法卸载。
UninstallOpenError=文件“%1”无法打开。无法卸载
UninstallUnsupportedVer=卸载日志文件“%1”的格式不受此版本的卸载程序支持。无法卸载
UninstallUnknownEntry=在卸载日志中遇到了未知条目 (%1)
ConfirmUninstall=您确定要完全删除 %1 及其所有组件吗？
UninstallOnlyOnWin64=此安装程序只能在 64 位 Windows 上卸载。
OnlyAdminCanUninstall=此安装程序只能由具有管理员权限的用户卸载。
UninstallStatusLabel=正在从您的电脑中删除 %1，请稍候。
UninstalledAll=%1 已成功从您的电脑中删除。
UninstalledMost=%1 卸载完成。%n%n某些元素无法删除。这些可以手动删除。
UninstalledAndNeedsRestart=为了完成 %1 的卸载，您的电脑必须重新启动。%n%n您现在想重新启动吗？
UninstallDataQuestion=您要保留用户数据和配置文件吗？

; *** Uninstallation status messages
StatusUninstalling=正在卸载 %1...
StatusDeletingFiles=正在删除文件...
StatusDeletingIniEntries=正在删除 INI 条目...
StatusDeletingRegistryEntries=正在删除注册表项...
StatusDeleteIcons=正在删除快捷方式...
StatusCleaningUp=正在清理...

; *** Uninstallation errors
ErrorOccurred=在卸载过程中发生错误。
CantTestPriorToUninstall=无法在卸载前测试。
AccessViolPriorToUninstall=在卸载前发生访问冲突。
BadUninstallCode=卸载程序代码已损坏。
ErrorRegisteringUninstaller=注册卸载程序时出错。
FileDoesntExist=文件“%1”不存在。
CustomMessages=

[CustomMessages]
chinesesimplified.CreateDesktopIcon=创建桌面快捷方式(&D)
chinesesimplified.AdditionalIcons=附加快捷方式:
chinesesimplified.LaunchProgram=启动
chinesesimplified.UninstallProgram=卸载

english.CreateDesktopIcon=Create a &desktop shortcut
english.AdditionalIcons=Additional shortcuts:
english.LaunchProgram=Launch
english.UninstallProgram=Uninstall
