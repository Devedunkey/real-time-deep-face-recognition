
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import sys
import io
import datetime


def update_time_sheet(name_member):
    sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

    credentials = ServiceAccountCredentials.from_json_keyfile_name('door_entry_key.json', scope)

    gc = gspread.authorize(credentials)

    # 현재시간 가져오기
    now = datetime.datetime.now()
    wks = gc.open('entry_sheet')
    sheet = wks.worksheet('sheet1')

    # 이름찾기
    name_wks = gc.open('allowed_list')
    sheet_name = name_wks.worksheet('sheet1')


    # 영어 이름으로 검색 First Name으로 검색
    list_of_name_cells = sheet_name.findall(name_member)
    row_name_updated = 0
    col_name_updated = 0
    for cell in list_of_name_cells:
        row_name_updated = cell.row
        col_name_updated = cell.col

    # 해당 이름 없다면 Stop
    if row_name_updated == 0 and col_name_updated == 0:
        return

    # 한글이름 가져오기
    korean_name = sheet_name.cell(row_name_updated, col_name_updated - 2).value


    # 출퇴근 등록여부 확인
    is_registered = sheet_name.cell(row_name_updated, col_name_updated + 5).value

    if is_registered != "O":
        return

    # Get a list of sheet value by name
    list_of_cells = sheet.findall(korean_name)
    row_updated  = 0
    for cell in list_of_cells:
        row_updated = cell.row


    # search week of day
    weekDayList = ['월', '화', '수', '목', '금', '토', '일']
    wkofday = datetime.datetime.today().weekday()

    # Create search text
    #now = datetime.datetime.now()
    date_search =  (str(now.year)[2:]) + '년 ' + str(now.month) + '월 ' + str(now.day) + '일 ' + str(weekDayList[wkofday])

    # Get a list of date
    list_of_date_cells = sheet.findall(date_search)
    column_updated  = 0
    for cell in list_of_date_cells:
        column_updated = cell.col

    day_half = '오전'
    hour = now.hour
    if hour > 11:
        hour = hour % 12
        day_half = '오후'

    # 출근시간 Cell이 비어있는지 확인
    entry_value = sheet.cell(row_updated, column_updated).value

    if entry_value == '':
        # Insert entry value
        sheet.update_cell(row_updated, column_updated, day_half + ' ' + str(hour) + ':' + str(now.minute))
    else:
        # Insert exit value
        sheet.update_cell(row_updated + 1, column_updated, day_half + ' ' + str(hour) + ':' + str(now.minute))