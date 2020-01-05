using ProgressMeter

data = [
    Dict(:file => "/host/data/data-salary.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap04/input/data-salary.txt"),
    Dict(:file => "/host/data/data-attendance-1.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap05/input/data-attendance-1.txt"),
    Dict(:file => "/host/data/data-attendance-2.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap05/input/data-attendance-2.txt"),
    Dict(:file => "/host/data/data-attendance-3.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap05/input/data-attendance-3.txt"),
    Dict(:file => "/host/data/data-rental.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-rental.txt"),
    Dict(:file => "/host/data/data-aircon.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap07/input/data-aircon.txt"),
    Dict(:file => "/host/data/data-salary-2.txt", :url => "https://raw.githubusercontent.com/MatsuuraKentaro/RStanBook/master/chap08/input/data-salary-2.txt"),

]

@showprogress 1 "downloading..." for d in data
    download(d[:url], d[:file])
end



